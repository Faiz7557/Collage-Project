@extends('layouts.main')

@section('content')
<div class="container mx-auto px-4 py-6">
    <!-- Judul Sederhana -->
    <div class="row mb-5">
        <div class="col-12">
            <h1 class="h3 fw-bold text-dark mb-2">
                <i class="fas fa-cog me-2 text-success"></i>
                Pengaturan Akun
            </h1>
            <p class="text-muted mb-0">Kelola informasi profil, password, dan akun Anda di sini.</p>
        </div>
    </div>

    <!-- Header Profil Sederhana -->
    <div class="row mb-5">
        <div class="col-12">
            <div class="card shadow-sm border-0">
                <div class="card-body p-4">
                    <div class="row align-items-center">
                        <div class="col-md-3 text-center mb-3 mb-md-0">
                            <div class="profile-avatar mb-2">
                                @if($user->foto)
                                    <img src="{{ asset('storage/users/' . $user->foto) }}" alt="{{ $user->name }}" class="rounded-circle img-fluid" style="width: 80px; height: 80px; object-fit: cover;">
                                @else
                                    <div class="rounded-circle bg-light d-flex align-items-center justify-content-center" style="width: 80px; height: 80px;">
                                        <i class="fas fa-user text-secondary fs-3"></i>
                                    </div>
                                @endif
                            </div>
                            <span class="badge bg-success rounded-pill px-3 py-1">
                                {{ ucfirst($user->peran ?? 'User ') }}
                            </span>
                        </div>
                        <div class="col-md-9">
                            <h4 class="mb-2">{{ $user->name }}</h4>
                            <p class="text-muted mb-1">Email: {{ $user->email }}</p>
                            <p class="text-muted mb-0">Bergabung sejak: {{ $user->created_at->format('d M Y') }}</p>
                            @if(session('status') === 'profile-updated' || session('status') === 'password-updated')
                                <div class="alert alert-success alert-dismissible fade show mt-3" role="alert">
                                    <i class="fas fa-check-circle me-2"></i>
                                    {{ session('status') === 'profile-updated' ? 'Profil berhasil diperbarui.' : 'Password berhasil diperbarui.' }}
                                    <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
                                </div>
                            @endif
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <div class="row g-4">
        <!-- Update Profile Information -->
        <div class="col-lg-4 col-md-6">
            <div class="card shadow-sm border-0 h-100">
                <div class="card-header bg-light border-0 py-3">
                    <h5 class="card-title mb-0 fw-semibold text-dark">
                        <i class="fas fa-user-edit me-2 text-success"></i>
                        Informasi Profil
                    </h5>
                </div>
                <div class="card-body p-4">
                    @include('profile.partials.update-profile-information-form')
                </div>
            </div>
        </div>

        <!-- Update Password -->
        <div class="col-lg-4 col-md-6">
            <div class="card shadow-sm border-0 h-100">
                <div class="card-header bg-light border-0 py-3">
                    <h5 class="card-title mb-0 fw-semibold text-dark">
                        <i class="fas fa-lock me-2 text-success"></i>
                        Ubah Password
                    </h5>
                </div>
                <div class="card-body p-4">
                    @include('profile.partials.update-password-form')
                </div>
            </div>
        </div>

        <!-- Delete Account -->
        <div class="col-lg-4 col-md-12">
            <div class="card shadow-sm border-0 h-100">
                <div class="card-header bg-light border-0 py-3">
                    <h5 class="card-title mb-0 fw-semibold text-dark">
                        <i class="fas fa-exclamation-triangle me-2 text-warning"></i>
                        Hapus Akun
                    </h5>
                </div>
                <div class="card-body p-4">
                    @include('profile.partials.delete-user-form')
                </div>
            </div>
        </div>
    </div>
</div>

<style>
    /* CSS Sederhana dan Minimalis untuk Halaman Settings */
    
    /* Card Umum: Sederhana dengan Shadow Ringan */
    .card {
        transition: all 0.2s ease;
        border-radius: 10px;
    }
    
    .card:hover {
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
        transform: translateY(-2px);
    }
    
    .card-header {
        background-color: #f8f9fa !important;
    }
    
    /* Form Inputs: Sederhana dengan Focus State */
    .form-control, .form-control:focus {
        border: 1px solid #dee2e6;
        border-radius: 6px;
        padding: 10px 12px;
        font-size: 0.95rem;
        transition: border-color 0.15s ease-in-out, box-shadow 0.15s ease-in-out;
    }
    
    .form-control:focus {
        border-color: #28a745;
        box-shadow: 0 0 0 0.2rem rgba(40, 167, 69, 0.25);
        outline: none;
    }
    
    .form-control::placeholder {
        color: #adb5bd;
    }
    
    /* Buttons: Konsisten dengan Tema Hijau */
    .btn-primary {
        background-color: #28a745;
        border-color: #28a745;
        border-radius: 6px;
        padding: 8px 16px;
        font-weight: 500;
        transition: all 0.15s ease;
    }
    
    .btn-primary:hover {
        background-color: #218838;
        border-color: #1e7e34;
        transform: translateY(-1px);
    }
    
    .btn-danger {
        background-color: #dc3545;
        border-color: #dc3545;
        border-radius: 6px;
        padding: 8px 16px;
        font-weight: 500;
        transition: all 0.15s ease;
    }
    
    .btn-danger:hover {
        background-color: #c82333;
        border-color: #bd2130;
        transform: translateY(-1px);
    }
    
    .btn-secondary {
        background-color: #6c757d;
        border-color: #6c757d;
        border-radius: 6px;
        padding: 8px 16px;
        font-weight: 500;
        transition: all 0.15s ease;
    }
    
    .btn-secondary:hover {
        background-color: #5a6268;
        border-color: #545b62;
        transform: translateY(-1px);
    }
    
    /* Error Messages: Sederhana */
    .invalid-feedback {
        font-size: 0.875rem;
        color: #dc3545;
        display: block;
        margin-top: 0.25rem;
    }
    
    /* Badge Role: Sederhana */
    .badge {
        font-size: 0.8rem;
        padding: 6px 12px;
    }
    
    /* Alert: Sederhana */
    .alert {
        border-radius: 6px;
        border: none;
        font-size: 0.9rem;
    }
    
    /* File Upload: Sederhana (untuk partial foto) */
    .form-file-label {
        border: 2px dashed #dee2e6;
        border-radius: 6px;
        padding: 40px 10px;
        text-align: center;
        cursor: pointer;
        transition: border-color 0.15s ease;
        background-color: #f8f9fa;
    }
    
    .form-file-label:hover {
        border-color: #28a745;
        background-color: #e9ecef;
    }
    
    /* Modal: Sederhana (jika digunakan di partial) */
    .modal-content {
        border-radius: 10px;
        border: none;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
    }
    
    .modal-header {
        border-bottom: 1px solid #dee2e6;
        background-color: #f8f9fa;
        border-radius: 10px 10px 0 0;
    }
    
    /* Responsif: Pastikan Mobile-Friendly */
    @media (max-width: 768px) {
        .container {
            padding-left: 1rem;
            padding-right: 1rem;
        }
        
        .card-body {
            padding: 1.25rem;
        }
        
        .row.g-4 > div {
            margin-bottom: 1rem;
        }
        
        h1 {
            font-size: 1.75rem;
        }
        
        .profile-avatar img, .profile-avatar div {
            width: 60px !important;
            height: 60px !important;
        }
    }
    
    /* Animasi Minimalis */
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    
    .card {
        animation: fadeIn 0.3s ease-out;
    }
</style>
@endsection